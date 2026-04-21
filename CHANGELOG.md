# [0.4.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.3...v0.4.0) (2026-04-20)


### Features

* EmbeddingClient and Providers in omop-emb, fix non-canonical model naming without tags ([#6](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/6)) ([1aaf8e7](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/1aaf8e732c12eadbe79c288118d9ec8ec2e25575))

## [0.3.3](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.2...v0.3.3) (2026-04-15)


### Bug Fixes

* Limit the number of concepts retrieved through EmbeddingConceptFilter for unified interface ([#5](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/5)) ([4bfa86b](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/4bfa86b4808b5f2f01297bee76474fc1aaf5ad9c))

## [0.3.2](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.1...v0.3.2) (2026-04-12)


### Bug Fixes

* correctly normalise all distances to [0,1] similarities ([5657aef](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5657aefb41f59cdfb405c21fdd640b1ee601f74d))

## [0.3.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.0...v0.3.1) (2026-04-12)


### Bug Fixes

* cleanup residuals ([dc890f7](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/dc890f7b3fd6b838a49eeab739e579034e6cb3bd))

# [0.3.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.5...v0.3.0) (2026-04-12)


### Features

* Local metadata storage ([#2](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/2)) ([1aa8da5](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/1aa8da55c4790abe7e2a4540fc3c8cdad2f88470))

## [0.2.5](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.4...v0.2.5) (2026-04-03)


### Bug Fixes

* commit pgvector extension ([5fd66b6](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5fd66b6ba95907efd76d39ed7bebde34a9f08708))
* list existing models correctly, create vector extension only for pgvector ([47618df](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/47618dfaa1e7400485d07fede548a590a617bc3e))

## [0.2.4](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.3...v0.2.4) (2026-04-02)


### Bug Fixes

* trigger release for latest main changes ([a188e0c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/a188e0cdf2c33d4869fb84daaa6dc2a2408e2206))

## [0.2.3](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.2...v0.2.3) (2026-04-01)


### Bug Fixes

* redirect omop-llm dependency to pypi, adaptations to CI/CD ([03315a2](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/03315a26d507e9eb5caa54ffdbaa8a3aff740836))

## [0.2.2](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.1...v0.2.2) (2026-04-01)


### Bug Fixes

* trigger initial 0.x release ([45a8308](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/45a830801ccc92cc5bf806a4109f3a3447a98519))

## [0.2.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.0...v0.2.1) (2026-04-01)


### Bug Fixes

* trigger PyPI publish after OIDC config ([2fb4b40](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/2fb4b40c4d9221ceac1ae1f3a25f7059380b53bd))

# [0.2.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.1.0...v0.2.0) (2026-04-01)


### Bug Fixes

* pull newest omop-llm ([c3fb805](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c3fb8050d84804f48a44966e5d4271465485652d))
* remove dupblicat optional-dep ([c7a58c1](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c7a58c16078615a42e5dc6787e0abb44449ab273))
* Remove duplicate "scripts" key after PR ([9d8369d](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/9d8369d9c54aecddb39efcefada186748eb6ff0f))


### Features

* diverse interface for embedding backends ([3d696dc](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/3d696dc906ac7a94ee7464b82459b0a9b9db3ed2))
