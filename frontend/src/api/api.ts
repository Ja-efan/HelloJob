import axios from "axios";

const baseURL = import.meta.env.DEV ? "" : "https://k12b105.p.ssafy.io";

export const api = axios.create({
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const requestInfo = {
      url: error.config?.url,
      method: error.config?.method,
      headers: error.config?.headers,
      data: error.config?.data,
      params: error.config?.params,
    };

    if (error.response) {
      const errorStatus = error.response.status;
      const errorData = error.response.data;
      const requestUrl = error.config.url;

      // 응답 정보
      const errorResponse = {
        requestUrl: requestUrl,
        status: errorStatus,
        statusText: error.response.statusText,
        data: errorData,
        headers: error.response.headers,
      };

      switch (errorStatus) {
        case 400:
          console.error(`${errorStatus} 오류 입니다`, errorResponse);
          break;
        case 401:
          console.error(
            `${errorStatus} Unauthorized: 인증 오류`,
            errorResponse
          );
          alert("인증이 만료되었습니다");
          window.location.replace("/login");
          break;
        case 403:
          console.error(`${errorStatus} Forbidden: 권한 오류`, errorResponse);
          break;
        case 404:
          console.error(
            `${errorStatus} Not Found: 요청한 리소스가 서버에 없음`,
            errorResponse
          );
          break;
        case 413:
          console.error(`${errorStatus} 파일크기가 너무 커요`, errorResponse);
          break;
        case 422:
          console.error(
            `${errorStatus} Unprocessable Entity: 요청은 유효하나 처리 실패`,
            errorResponse
          );
          break;
      }
    } else if (error.request) {
      console.error("네트워크 에러:", error.request);
      console.error("요청 정보:", requestInfo);
    } else {
      console.error("클라이언트 에러", error.message);
      console.error("요청 정보:", requestInfo);
    }

    return Promise.reject(error);
  }
);
