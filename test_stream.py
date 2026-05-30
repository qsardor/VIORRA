import litert_lm
e = litert_lm.Engine('C:/Users/Sardor/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/a4a831c060880f3733135ad22f10e0e9f758f45d/gemma-4-E2B-it.litertlm', backend=litert_lm.interfaces.GPU())
c = e.create_conversation()
stream = c.send_message_async('Explain how gravity works in 10 words.')
print('---')
count = 0
for chunk in stream:
    print(f"Chunk {count}:", repr(chunk['content'][0]['text']))
    count += 1
    if count > 5:
        break
