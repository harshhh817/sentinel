# Chaincode-as-a-service image: built on the host, the peer only connects to it.
# (Fabric 2.5's in-peer docker build uses a legacy Docker API that current Docker
# Desktop engines refuse; CCaaS sidesteps that entirely.)
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 GOFLAGS=-mod=vendor go build -o /auditcontract .

FROM alpine:3.20
COPY --from=build /auditcontract /auditcontract
# test-network's deployCCAAS sets CHAINCODE_SERVER_ADDRESS and CORE_CHAINCODE_ID_NAME;
# contractapi's Start() switches to server mode when they are present.
ENV CHAINCODE_SERVER_ADDRESS=0.0.0.0:9999
EXPOSE 9999
CMD ["/auditcontract"]
