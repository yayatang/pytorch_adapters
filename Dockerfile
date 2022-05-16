FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:latest
RUN pip3 install --user torch torchvision imgaug scikit-image


# docker build -t gcr.io/viewo-g/modelmgmt/resnet:0.0.1 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/modelmgmt/resnet:0.0.1
