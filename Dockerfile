# Build Image
FROM nvidia/cuda:7.5-devel-ubuntu14.04 AS build

RUN apt-get update; exit 0
RUN apt-get install -y --no-install-recommends git

COPY src .
RUN sh build.sh

# Runtime image
FROM nvidia/cuda:7.5-runtime-ubuntu14.04

COPY --from=build main main

CMD [ "./main" ]

