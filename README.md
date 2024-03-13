# Code generation app

## Quickstart

Ensure you have the GPU container toolkit if you want to use a local LLM (see Docker
compose section).

Also make sure that your OpenAI API key is exported to the default location on your host
machine: OPENAI_API_KEY. Alternatively, if you do not wish to do that, you can also
directly enter the API key into the docker compose.yaml on line 6, replacing
`${OPENAI_API_KEY}`.

From the llm_code_app directory run:

```shell
docker compose up
```

Wait for the containers to build and run. This can take a few minutes. The app will
be running at `localhost:8084`

## About the repository

Normally, I would squash, merge then delete feature branches. I have not done that to
allow looking further into each commit if so desired.

## Backend features

Uses FastAPI with two post endpoints: generate and adjust. The first is used on the
first 'turn.' The second is used when the user has already generated some code and is
providing feedback to adjust it.

### Feedback Loop

The adjust endpoint is the backend portion of this essential component.

### Custom LLM integration

I have already explicitly added a custom LLM: Hermes. The LLMBackend class structure is
very straightforward, so adding further models should be rather simple. For models
following the Huggingface structure, you only have so subclass the `HuggingfaceBackend`
class and implement the `encode_prompt` and `decode_output` methods. These are used to
format and tokenize the input prompt and decode outputs based on model-specific needs.

If you instead want to include a model with a different structure, you will need to
implement the base `LLMBackend` class.

### Containerization

The backend is packaged in a docker container and is only visible to the frontend
container.

### Prompt Security

Utilizes specific system prompts and prompt layout, along with low temperature to
ensure stable generation. Additionally, error returns are flagged. However, there is
always the possibility something gets through, and in that case further measures, like
doing a double query (Is this code?) would be advisable, as well as more thorough
pre-processing to check for attacks.

## Frontend features

I am not a frontend engineer! But I can work with it to some extent. Here I've used the
Reflex framework in python, which basically just compiles python into javascript.

### Web interface

A core component. The frontend does this well enough. I have created 'sessions' that
users can create and swap between, each with their own language and model. This keeps
some ambiguity from creeping in (switching between models and languages after some
generations) and makes it easier to keep track of.

### Feedback loop

The feedback loop happens automatically, allowing users to send multiple messages, with
each message after the first being feedback on the previously generated code snippet.

### Custom LLM integration

Most of this work happens on the backend, but the frontend provides a selector for model
name, which is then converted to the model id and sent to the backend API.

### Containerization

As with the backend, containerized. Reflex uses a different project structure and python
version, meaning splitting the frontend and backend into separate containers has
multiple motivations. It communicates with the backend via the default docker compose
network, but since the browser needs to communicate with the Reflex "backend," the ports
for both the frontend and Reflex "backend" are forwarded through.

### Prompt Security

Simple checks for very long or short prompts, and boxing the model and language choices
also helps. Errors are clearly marked.

### Snippet Management

Part of this is implemented in the session system, namely creating new sessions along
with looking at and switching to old sessions. Snippet and session deletion aren't
implemented, but they are a fairly straightforward extrapolation.

### Snippet Testing System

Not implemented. I know there is an OpenAI endpoint for this, I just didn't wish to use
it for this project. Local interpretation could also be used, but is riskier.

### A note about data structures

Reflex supports Pydantic data models and dicts, or so they say. I simply could not get
even the simplest to compile in this project, so I had to revert to using primitives,
lists and tuples. My apologies, I tried to make it as readable as possible.

## Docker compose

### Building and running the images

From the llm_code_app directory, simply run

```shell
docker compose up
```

That's it. GPU, and by extension the Huggingface models, will not be available unless
you follow the directions below.

Building the backend docker image will take some time as the torch and nvidia packages
are quite large and take a while to download. The frontend docker also needs to compile
and export the python into javascript, which can take a few minutes.

### Using GPU inside the compose

The provided compose file will only work if GPU containers are enabled. This will
require some work on your part, if you haven't done it already. If you are running
Windows, there is a built-in container-level GPU functionaliy. I don't use Windows, but
you can read more here: `https://docs.docker.com/desktop/gpu/`. I can't give any
guarantees on this working for Windows.

Now, for Unix systems, you will need to install the Nvidia container toolkit and
configure the docker daemon accordingly: `https://docs.nvidia.com/datacenter/
cloud-native/container-toolkit/latest/install-guide.html`.

And the rest is done for you in the `compose.yaml`. Only the backend can access the gpu.

### Volumes

I haven't used any volumes in any of the docker files. If you add in volumes, the
overhead for loading models for the first time could be significantly reduced (esp. wrt.
downloading from the hub).

## Known Issues

I am lazily instantiating models in the backed, meaning large models may need to be
downloaded and will need to be loaded into memory after the API call. This will time out
the call on the frontend, which means you'll just have to repeat the call when the model
has finished loading. May need to refresh the page, too.

Language and model dropdowns sometimes save state across reloads separate the underlying
variable state. This results in erroneous error boxes. Just change the value and
change it back in each box.

Adding a new session or switching to an old session when the active session is blank
can result in fatal errors. Clear website data and reload.

Sometimes the frontend loses connection to the Reflex "backend" for no apparent reason.
Restarting the docker fixes this.
