from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pandasai import SmartDataframe
import pandas as pd
import chainlit as cl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

llm = ChatOpenAI()

df = pd.read_csv("purchase_order_light.csv")

agent = SmartDataframe(df, config={"llm": llm})


@cl.on_message
async def main(message: cl.Message):
    result = await cl.make_async(agent.chat)(
        {
            "query": message.content,
        }
    )

    print("RESULT: ", result)

    await cl.Message(content=result).send()


if __name__ == "__main__":

    class Query(BaseModel):
        message: str

    app = FastAPI()

    @app.post("/query")
    def receive_message(body: Query):
        try:
            result = agent.chat(
                {
                    "query": body.message,
                }
            )

            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
