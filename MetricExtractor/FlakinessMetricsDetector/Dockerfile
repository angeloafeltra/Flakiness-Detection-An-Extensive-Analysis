FROM eclipse-temurin:17-jdk-jammy

WORKDIR /FlakinessMetricsDetector
COPY .mvn ./.mvn
COPY mvnw pom.xml ./
COPY lib ./lib
COPY src ./src


RUN mkdir -p ./spazioCondiviso
RUN mkdir -p ./spazioCondiviso/Dataset
RUN mkdir -p ./spazioCondiviso/MetricsDetector
RUN mkdir -p ./spazioCondiviso/Repository



RUN ./mvnw dependency:resolve
RUN ./mvnw package

CMD ["java", "-jar", "./target/FlakinessMetricsDetector-0.0.1-SNAPSHOT.jar"]
