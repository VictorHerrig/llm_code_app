import reflex as rx

config = rx.Config(
    app_name="codegen_frontend",
    backend_port=8082,
    frontend_port=8084,
    api_url='http://localhost:8082',
)
