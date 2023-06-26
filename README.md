# dropout-similar-images
Here is a small project to filter out similar images from a directory.



#### Building a container
```shell

cd /path/to/the/repo
docker build -t dataset-proc-img -f docker/Dockerfile .
```

```shell
docker run \
    --rm \
    -v /path/to/dataset:/dataset \
    --name dataset-proc \
    dataset-proc-img \
    remove-similar-images \
    --dataset-path /dataset \
    --output-dir-path /dataset/selected \
    --min-contour-area-diff 30000 \
    --min-imsize-percentile 75 \
    --min-imsize-scale 0.5 \
    --black-mask 10 10 10 10 \
    --gaussian-blur-radiuses 11 15 \
    --min-contour-area 1000 \
    --save-data-analysis-plots-to /dataset/data-analysis
```
