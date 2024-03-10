## Docker
### Building the image
From the `backend` directory (i.e., here), run:

```shell
docker build -t <image_name>:<tag> .
```

### Run the container
#### Without GPU
The Docker container will not be able to see the onboard GPU devices without extra work,
so only OpenAI can be used by default. In order to use OpenAI, you will have to provide
your own API key. Assuming you have the key saved in the default environment variable of
`OPENAI_API_KEY`, the following will run the server at `localhost:8083` with access to
OpenAI: 
```shell
docker run -p 8083:80 -e OPENAI_API_KEY=$OPENAI_API_KEY <image_name>:<tag>
```

#### With GPU
You will first need to install the Nvidia container toolkit and configure, as described
in the README in the top-level directory. Then the process for for running the docker is
very similar. If you want to pass all gpus to the container with the same settings as
above, run the command:
```shell
docker run -p 8083:80 -e OPENAI_API_KEY=$OPENAI_API_KEY \
--gpus all <image_name>:<tag>
```
Or you can specify the gpus to be passed if on a multi-GPU cluster.
