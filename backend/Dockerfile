# 빌드 스테이지
FROM gradle:8.13-jdk17 AS build
# 작업 디렉토리 설정
WORKDIR /app
# Gradle 설정 파일 복사
COPY gradle gradle
COPY gradlew ./
COPY build.gradle settings.gradle ./
# 소스 코드 복사
COPY src src
# 테스트 제외하고 빌드
RUN ./gradlew build -x test

# 실행 스테이지
FROM openjdk:17-slim
WORKDIR /app
# 빌드된 jar 복사
COPY --from=build /app/build/libs/*.jar app.jar
# 포트 설정
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]