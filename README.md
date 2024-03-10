# Code generation app
## Docker compose
### Building and running the images
From the llm_code_app directory, simply run
```shell
docker compose up
```
That's it. GPU, and by extension the Huggingface models, will not be available unless
you follow the directions below.

Building the backend docker will take some time as the torch and nvidia packages are
quite large and take a while to download.

### Using GPU inside the compose
The provided compose file will only work if GPU containers are enables. This will
require some work on your part, if you haven't done it already. If you are running
Windows, there is a built-in container-level GPU functionaliy. I don't use Windows, but
you can read more here: `https://docs.docker.com/desktop/gpu/`. I can't give any
guarantees on this working for Windows.

Now, for Unix systems, the process can happen on the compose level. First, you will need
to install the Nvidia container toolkit and configure the docker daemon accordingly:
`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
latest/install-guide.html`.

And the rest is done for you in the `compose.yaml`.

### Volumes
I haven't used any volumes in any of the docker files. If you add in volumes, the
overhead for loading models for the first time could be significantly reduced (esp. wrt.
downloading from the hub).
