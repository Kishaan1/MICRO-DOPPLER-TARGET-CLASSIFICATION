import urllib.request
import json

body = "time,amplitude\n" + "\n".join([f"{i*0.001},{i%5*0.1}" for i in range(51)])
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

payload = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="short_signal.csv"\r\n'
    f"Content-Type: text/csv\r\n\r\n"
    f"{body}\r\n"
    f"--{boundary}--\r\n"
).encode('utf-8')

req = urllib.request.Request(
    "http://127.0.0.1:5000/api/upload-signal",
    data=payload,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
print("Short signal upload success:", res["success"], "Target class:", res["target_class"], "Confidence:", res["confidence"])
