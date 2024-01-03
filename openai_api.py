import numpy as np
import asyncio
import logging
import time
from typing import List, Literal, Optional, Union

import chatglm_cpp
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModel
import torch

logging.basicConfig(level=logging.INFO,
                    format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")


class Settings(BaseSettings):
    model: str = ".\models\chatglm3\chatglm3-ggml-q4_0.bin"
    embeddingModel: str = ".\models\shibing624\\text2vec-base-chinese"
    num_threads: int = 0


settings = Settings()
embeddingTokenizer = AutoTokenizer.from_pretrained(settings.embeddingModel)
embeddingModel = AutoModel.from_pretrained(settings.embeddingModel)
pipeline = chatglm_cpp.Pipeline(settings.model)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
lock = asyncio.Lock()


def average_embeddings(embeddings):
    """
    计算嵌入向量列表的平均值。
    :param embeddings: 嵌入向量的列表。
    :return: 平均嵌入向量。
    """
    # 将嵌入向量列表转换为 NumPy 数组
    embeddings_array = np.array(embeddings)

    # 计算平均值
    avg_embedding = np.mean(embeddings_array, axis=0)

    return avg_embedding.tolist()


def split_into_parts(text, max_length):
    """
    将文本分割成多个部分，每部分长度不超过 max_length。
    :param text: 要分割的原始文本。
    :param max_length: 每部分文本的最大长度。
    :return: 分割后的文本列表。
    """
    # 确保文本是字符串
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    # 初始化变量
    parts = []
    current_part = ""

    # 按空格分割文本，简单处理
    for word in text.split():
        if len(current_part) + len(word) + 1 > max_length:
            parts.append(current_part)
            current_part = word
        else:
            current_part += " " + word

    # 添加最后一部分
    if current_part:
        parts.append(current_part)

    return parts


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "owner"
    permission: List = []


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "owner", "permission": []}],
                }
            ]
        }
    }


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "default-model"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice],
                   List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "default-model",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


class EmbeddingCreateParams(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class Embedding(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: str = "list"
    usage: dict


def stream_chat(messages, body):
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(
            delta=DeltaMessage(role="assistant"))],
    )

    for chunk in pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=settings.num_threads,
        stream=True,
    ):
        yield ChatCompletionResponse(
            object="chat.completion.chunk",
            choices=[ChatCompletionResponseStreamChoice(
                delta=DeltaMessage(content=chunk.content))],
        )

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(
            delta=DeltaMessage(), finish_reason="stop")],
    )


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
            for chunk in stream_chat(history, body):
                # yield control back to event loop for cancellation check
                await asyncio.sleep(0)
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        logging.info(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        logging.info(
            f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e


@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    logging.info(
        f'prompt: "{body.messages[-1].content}"')

    messages = [chatglm_cpp.ChatMessage(
        role=msg.role, content=msg.content) for msg in body.messages]

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)

    max_context_length = 512
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )

    logging.info(
        f'sync response: "{output.content}"')

    # 如果返回的结果长度小于1024，可能会导致进程结束，参考
    # https://github.com/li-plus/chatglm.cpp/issues/244
    if body.max_tokens < 1024:
        body.max_tokens = 1024

    prompt_tokens = len(pipeline.tokenizer.encode_messages(
        messages, max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(
        output.content, body.max_tokens))

    logging.info("Returning from function")
    return ChatCompletionResponse(
        object="chat.completion",
        choices=[ChatCompletionResponseChoice(
            message=ChatMessage(role="assistant", content=output.content))],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@app.post("/v1/embeddings")
def create_embeddings(params: EmbeddingCreateParams) -> CreateEmbeddingResponse:
    inputs = params.input if isinstance(params.input, list) else [params.input]
    embeddings = []
    total_tokens = 0

    for index, input_text in enumerate(inputs):
        if isinstance(input_text, (str, list)):
            # 分割文本
            # 减去2是为了 [CLS] 和 [SEP] token
            parts = split_into_parts(input_text, max_length=512-2)

            embeddings_for_this_input = []
            for part in parts:
                inputs = embeddingTokenizer(
                    part, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                total_tokens += inputs.input_ids.size(1)

                with torch.no_grad():
                    outputs = embeddingModel(**inputs)

                embedding = outputs.last_hidden_state.mean(
                    dim=1).squeeze().tolist()
                embeddings_for_this_input.append(embedding)

            # 对单个输入的所有部分进行平均或其他形式的合并
            final_embedding = average_embeddings(embeddings_for_this_input)
            embeddings.append(
                Embedding(embedding=final_embedding, index=index))

    return CreateEmbeddingResponse(
        data=embeddings,
        model=params.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id="gpt-3.5-turbo"), ModelCard(id="text-embedding-ada-002")])


uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
