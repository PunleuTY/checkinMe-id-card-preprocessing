# checkinMe-id-card-preprocessing

```

 BASE64=$(base64 -i sample_imgs/id2.jpg) && \
  curl -s -X POST http://127.0.0.1:8000/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$BASE64\"}" \
    | python3 -c "
  import sys, json, base64
  r = json.load(sys.stdin)
  open('/tmp/result.jpg','wb').write(base64.b64decode(r['segmented_image']))
  " && open /tmp/result.jpg && open sample_imgs/id2.jpg

  ```

```
  BASE64=$(base64 -i sample_imgs/id2.jpg) && \
  curl -s -X POST http://127.0.0.1:8000/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$BASE64\"}" \
    | python3 -c "
  import sys, json, base64
  r = json.load(sys.stdin)
  open('/tmp/final.webp','wb').write(base64.b64decode(r['processed_image']))
  " && open /tmp/final.webp && open sample_imgs/id2.jpg
```
