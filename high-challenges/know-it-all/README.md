# Deploy



```shell
docker build -t know_it_all_image -f ./Dockerfile .
docker run -it -p 7462:7462 -e OPENAI_API_KEY=your_key --name know_it_all know_it_all_image
```

