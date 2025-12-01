from fastapi import FastAPI
from langserve import add_routes
from chains.subject_chain import make_physics_chain, make_math_chain
from routers.health_router import router as health_router
from routers.data_ingest import router as data_ingest_router
from routers.rag import router as rag_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="EyePatch API")

    # middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # build chains once at startup
    physics_chain = make_physics_chain()
    math_chain = make_math_chain()

    # register LangServe routes
    add_routes(app, physics_chain, path="/physics")
    add_routes(app, math_chain, path="/math")

    # custom routers
    app.include_router(health_router)
    app.include_router(data_ingest_router)
    app.include_router(rag_router)
    return app
