import os
import json
import cv2
import numpy as np
import tornado.ioloop
import tornado.web
import extractor

def _decode_image(files: dict, field: str = "file"):
    if field not in files or not files[field]:
        return None

    img_bytes = files[field][0]["body"]
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


class ScanHandler(tornado.web.RequestHandler):
    """Handle POST /scan → run OCR and return JSON."""

    async def post(self):
        img = _decode_image(self.request.files)
        if img is None:
            self.set_status(400)
            return self.write({"error": "no image uploaded"})

        info, _ = extractor.process_frame(img)
        self.set_header("Content-Type", "application/json")
        self.write({"parsed": info})


def make_app(static_dir: str) -> tornado.web.Application:
    """Create the Tornado application with routes."""
    return tornado.web.Application(
        [
            (r"/scan", ScanHandler),
            # Serve index.html and any other static assets in the same folder
            (r"/(.*)", tornado.web.StaticFileHandler,
             {"path": static_dir, "default_filename": "index.html"}),
        ],
        autoreload=True,
    )


if __name__ == "__main__":
    app = make_app(static_dir=os.path.dirname(__file__))
    app.listen(8888)
    print("✅ DigiScan running at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()