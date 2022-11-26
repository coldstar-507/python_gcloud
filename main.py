import functions_framework, replicate, requests, flask, base64

from replicate.exceptions import ModelError

client = replicate.Client(api_token="3e352b505e7f33370501180343d5443169fe1cce")

model = client.models.get('stability-ai/stable-diffusion')


@functions_framework.http
def imageGenerationRequest(request: flask.Request):
    decodedJson = request.get_json()
    prompt = decodedJson["prompt"]
    try:
        outputs = model.predict(prompt=prompt, width=512, height=512, num_outputs=1, num_inference_steps=10, guidance_scale=7.5)
        for out in outputs:
            with open(prompt + '.png', 'wb') as handle:
                response = requests.get(out, stream=True)

                if not response.ok:
                    print(response)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)

            return 'OK'

    except ModelError:
        print("Handle exception here my good friend")
        return 'OK'



@functions_framework.http
def imageGenerationRequest2(request: flask.Request):
    listOfPrompts = request.get_json()
    for prompt in listOfPrompts:
        try:
            out = model.predict(prompt=prompt, width=512, height=512, num_outputs=1, num_inference_steps=10, guidance_scale=7.5)
            response = requests.get(out[0])
            if response.ok:
                return flask.jsonify(image=base64.b64encode(response.content).decode("utf-8"), prompt=prompt)
        except ModelError:
            print("Not safe for work content")
            continue

    return "You are very unlucky", 500

