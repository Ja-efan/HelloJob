# Node.js 경량 이미지
FROM node:22.3.0-alpine
# 작업 디렉토리
WORKDIR /app
# package.json 파일 복사
COPY package*.json ./
# 의존성 설치
RUN npm install --legacy-peer-deps
# 소스 코드 복사
COPY . .
# 프로젝트 빌드
RUN npm run build
# 포트 설정
EXPOSE 5173
# 실행 명령어 수정 - npm start 대신 npm run dev 사용
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
