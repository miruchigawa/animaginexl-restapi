import os
import uvicorn
import sqlite3 as sql
from uuid import uuid4
from queue import Queue 
from threading import Thread 
from PIL import Image
from datetime import datetime 
from fastapi import FastAPI
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
    timestamp = datetime.now()
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
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_path():
    return { "status": "success", "message": "Welcome to AnimagineXL restapi", "issue": "https://github.com/miruchigawa/animaginexl-restapi/issues", "author": ["miruchigawa <moe@miwudev.my.id>" ]}

@app.get("/api/v1/txt2img")
def text_to_image(prompt: str = None,
                  neg_prompt: str = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
                  width:int = 1024,
                  height:int = 1024,
                  guidance_scale: int = 7,
                  steps:int = 28,
                  sampler: str = "DPM++ 2M Karras"):
    if not prompt or prompt == "":
        return {"status": "failed", "message": "Missing prompt!"}
    if queue.full():
        return { "status": "failed", "message": "Queue full, try again in 30 seconds." }
    
    connection = create_connection()
    cursor = create_cursor(connection)
    uid = str(uuid4())
    task = {
        "uid": uid,
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "sampler": sampler
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
    """, (uid, "on_progress", prompt, neg_prompt, steps, width, height, guidance_scale, sampler, ""))
    connection.commit()
    connection.close()

    queue.put(task)
    size_now = queue.qsize()
    
    return { "status": "on_progress", "uid": uid, "queue number": f"{size_now}/{queue.maxsize}"}

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
