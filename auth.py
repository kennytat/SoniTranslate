from fastapi import FastAPI, HTTPException, Form, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import asyncio
import sqlite3
from passlib.hash import bcrypt
import uvicorn
from itsdangerous import URLSafeSerializer
import aiosqlite
import gradio as gr

root = FastAPI()
# Secret key for session management
SECRET_KEY = "your-secret-key"
serializer = URLSafeSerializer(SECRET_KEY)
root.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
root.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Function to create SQLite connection
async def create_connection():
    return await aiosqlite.connect('db/auth.db')

async def init_database():
  conn = await create_connection()
  cursor = await conn.cursor()
  await cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
        )
    ''')
  await conn.commit()
  await conn.close()

# Function to fetch user ID by username
async def get_user_id(username):
    conn = await create_connection()
    cursor = await conn.cursor()
    await cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = await cursor.fetchone()
    await conn.close()
    return row
  
# Dependency to check if the user is logged in
def is_authenticated(request: Request):
    token = request.cookies.get("token")
    # print('is_authenticated:', token)
    if token:
        username = serializer.loads(token)
        user_id = asyncio.run(get_user_id(username))
        # print('is_authenticated:', user_id[0])
        if user_id:
            return user_id[0]
    return None
  
# Routes
@root.get("/", response_class=HTMLResponse)
async def home(request: Request):
    token = request.cookies.get("token")
    if token:
        username = serializer.loads(token)
        # Check if user exists in the database (session management)
        conn = await create_connection()
        cursor = await conn.cursor()
        await cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = await cursor.fetchone()
        if row:
            return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")
  

@root.get("/signup", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})
  
@root.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    hashed_password = bcrypt.hash(password)
    try:
        conn = await create_connection()
        cursor = await conn.cursor()
        await cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        await conn.commit()
        await conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return RedirectResponse(url="/", status_code=303)

@root.get("/login", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
  
@root.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = await create_connection()
    cursor = await conn.cursor()
    await cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = await cursor.fetchone()
    await conn.close()
    if row and bcrypt.verify(password, row[0]):
        token = serializer.dumps(username)
        response = RedirectResponse(url="/app", status_code=303)
        response.set_cookie(key="token", value=token)
        return response
    error = "Wrong username or password"
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@root.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("token")
    return response

# @root.get("/app", response_class=HTMLResponse)
# async def app(request: Request, is_auth: bool = Depends(is_authenticated)):
#     if not is_auth:
#         return RedirectResponse(url="/login")
    # return templates.TemplateResponse("app.html", {"request": request})

# # Define a catch-all route to handle requests for non-existing routes
# @root.get("/{path:path}")
# async def catch_all(path: str, request: Request):
#     print("path:", path, request)
#     # Check if the requested path matches any existing routes
#     if path == "app":
#         return RedirectResponse(url="/app", status_code=303)
#     elif path not in root.routes:
#         # Redirect to the login page
#         return RedirectResponse(url="/login")
#     else:
#         # Return an HTTP 404 error if the requested route does not exist
#         raise HTTPException(status_code=404, detail="Not Found")

def greet(request: gr.Request):
    return f"Welcome to Gradio:::"
  
with gr.Blocks() as main_demo:
    m = gr.Markdown("Welcome to Gradio!")
    gr.Button("Logout", link="/logout")
    main_demo.load(greet, None, m)

root = gr.mount_gradio_app(root, main_demo, path="/app", auth_dependency=is_authenticated)
    
if __name__ == "__main__":
    asyncio.run(init_database())
    uvicorn.run(root, host="127.0.0.1", port=8001)
