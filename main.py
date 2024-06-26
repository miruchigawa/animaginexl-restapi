import os
import torch
import uvicorn
import sqlite3 as sql
from uuid import uuid4
from queue import Queue 
from threading import Thread 
from PIL import Image
from datetime import datetime 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from model import load_pipeline, get_sampler, free

"""
SQlite3 Function Setup
"""
def create_connection(dbpath="database/database.sqlite3"):
    connection = sql.connect(dbpath)
    return connection

def create_cursor(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            uid TEXT NOT NULL, 
            status TEXT NOT NULL,
            prompt TEXT NOT NULL,
            neg_prompt TEXT NOT NULL,
            steps INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            guidance_scale INTEGER NOT NULL,
            sampler TEXT NOT NULL,
            result TEXT NOT NULL
        )
    """)
    return cursor

"""
Thread Worker Queue Setup
"""
queue = Queue(maxsize=10)

def worker():
    connection = create_connection()
    cursor = create_cursor(connection)

    while True:
        task = queue.get()
        generate_txt2img(task, cursor)
        connection.commit()
        queue.task_done()

Thread(target=worker, daemon=True).start()

"""
AnimagineXL Setup
"""
pipe = load_pipeline()

def generate_txt2img(task, cursor):
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    filename_output = f"static/generate-{task['uid']}-{timestamp}.png"
     
    scheduler = pipe.scheduler
    pipe.scheduler = get_sampler(task["sampler"], pipe.scheduler.config)()
   
    try:
        image = pipe(
            task["prompt"],
            negative_prompt=task["neg_prompt"],
            num_inference_steps=task["steps"],
            width=task["width"],
            height=task["height"],
            guidance_scale=task["guidance_scale"]
        ).images[0]
        image.save(filename_output)
        status = "success"
        result = filename_output
    except Exception as e:
        status = "failed"
        result = f"Failed to generate {e}"
    finally:
        pipe.scheduler = scheduler
        free()

    cursor.execute("""
        UPDATE queue SET status=?, result=? WHERE uid=?
    """, (status, result, task["uid"]))

"""
FastAPI Setup
"""
class Txt2ImgBody(BaseModel):
    prompt: str
    neg_prompt: str 
    width:int = 1024
    height:int = 1024
    guidance_scale: int
    steps:int
    sampler: str


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_path():
    """ The root path """

    return { "status": "success", "message": "Welcome to AnimagineXL restapi", "issue": "https://github.com/miruchigawa/animaginexl-restapi/issues", "author": ["miruchigawa <moe@miwudev.my.id>" ]}

@app.get("/api/v1/ping")
def ping_server():
    """ Returns server related information """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    
    """ 
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html#torch.cuda.mem_get_info
    """
    if device == "cuda":
        device = []
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vmem_used, vmem_total = torch.cuda.mem_get_info(i)
            device.append({ "id": i, "name": name, "memory_free": vmem_used, "memory_total": vmem_total })
    else:
        device = { "name": "cpu", }
    return { "status": "success", "message": { "response": "Nyaho", "device": device }}

@app.post("/api/v1/txt2img")
def text_to_image(body: Txt2ImgBody):
    if not body.prompt or body.prompt == "":
        return {"status": "failed", "message": "Missing prompt!"}
    if queue.full():
        return { "status": "failed", "message": "Queue full, try again in 30 seconds." }
    
    connection = create_connection()
    cursor = create_cursor(connection)
    uid = str(uuid4())
    task = {
        "uid": uid,
        "prompt": body.prompt,
        "neg_prompt": body.neg_prompt,
        "width": body.width,
        "height": body.height,
        "guidance_scale": body.guidance_scale,
        "steps": body.steps,
        "sampler": body.sampler
    }

    cursor.execute("""
        INSERT INTO queue (
            uid,
            status,
            prompt,
            neg_prompt,
            steps,
            width,
            height,
            guidance_scale,
            sampler,
            result
        ) VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
    """, (uid, "on_progress", body.prompt, body.neg_prompt, body.steps, body.width, body.height, body.guidance_scale, body.sampler, ""))
    connection.commit()
    connection.close()

    queue.put(task)
    size_now = queue.qsize()
    
    return { "status": "on_progress",
            "message": { 
                "uid": uid,
                "prompt": body.prompt,
                "neg_prompt": body.neg_prompt,
                "width": body.width,
                "height": body.height,
                "guidance_scale": body.guidance_scale,
                "steps": body.steps,
                "sampler": body.sampler,
                "queue_number": f"{size_now}/{queue.maxsize}"
                }
            }

@app.get("/api/v1/info")
def get_status_task(uid: str = None):
    if not uid:
        return { "status": "failed", "message": "Missing uid!" }

    connection = create_connection()
    cursor = create_cursor(connection)
    info = cursor.execute("""SELECT * FROM queue WHERE uid=?""", (uid,)).fetchone()
    connection.close()

    if not info:
        return { "status": "failed", "message": f"There has no uid start with {uid}" }

    if info[1] == "success":
        return {
            "status": "success",
            "uid": uid,
            "prompt": info[2],
            "neg_prompt": info[3],
            "steps": info[4],
            "width": info[5],
            "height": info[6],
            "guidance_scale": info[7],
            "sampler": info[8],
            "result": info[9]
        }
    
    return { "status": info[1],  "message": info[9]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
