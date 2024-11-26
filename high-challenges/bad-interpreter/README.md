# Deploy



```shell
docker build -t bad_interpreter_image -f ./Dockerfile .
docker run --rm -it -p 7464:7464 -e OPENAI_API_KEY=your_key --name bad_interpreter bad_interpreter_image
```

