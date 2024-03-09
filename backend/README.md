## Docker
### Building the image
From the `backend` directory (i.e., here), run:

```shell
docker build -t <image_name>:<tag> .
```

### Run the container
The bare Docker container will not be able to see the onboard GPU devices,
so only OpenAI can be used by default. In order to use OpenAI, you will
have to provide your own API key. Assuming you have the key saved in the
default environment variable of `OPENAI_API_KEY`, the following will run
the server at `localhost:8083` with access to OpenAI:

```shell
docker run -p 8083:80 -e OPENAI_API_KEY=$OPENAI_API_KEY <image_name>:<tag>
```
