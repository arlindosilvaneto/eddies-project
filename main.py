from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pandasai import SmartDataframe
import pandas as pd
import chainlit as cl
from fastapi import FastAPI, HTTPException
import uvicorn

load_dotenv

llm = ChatOpenAI()

df = pd.read_csv("purchase_orders.csv")

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
    app = FastAPI()

    @app.post("/query")
    async def receive_message(message: str):
        try:
            result = await agent.chat(
                {
                    "query": message,
                }
            )

            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
