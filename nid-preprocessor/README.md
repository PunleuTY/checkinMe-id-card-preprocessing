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
BASE64=$(base64 -i sample_imgs/id4.jpeg) && \
  curl -s -X POST http://127.0.0.1:8000/preprocess \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$BASE64\"}" \
  | python3 -c "
  import sys, json, base64
  r = json.load(sys.stdin)
  if 'processed_image' not in r:
      print('ERROR from server:', r)
      sys.exit(1)
  open('/tmp/final.webp','wb').write(base64.b64decode(r['processed_image']))
  print('OK - written to /tmp/final.webp')
  " && open sample_imgs/id4.jpeg && open /tmp/final.webp
```
