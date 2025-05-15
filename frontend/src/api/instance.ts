import axios from "axios";
import { useAuthStore } from "../store/userStore";

// 인증 없이 사용하는 API (즉, 쿠키 없이 사용하는 API)
export const plainApi = axios.create({
  baseURL: import.meta.env.BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// 인증이 필요한 API (즉, 쿠키가 필요한 API)
export const authApi = axios.create({
  baseURL: import.meta.env.BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

const invalidRefreshToken = () => {
  const { setIsLoggedIn, setIsLoginAlert } = useAuthStore.getState();
  setIsLoggedIn(false);
  setIsLoginAlert(true);
};

plainApi.interceptors.response.use(
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
            `${errorStatus} Unauthorized: 인증 오류 - 인증 과정이 필요한 API를 사용하면 안되는 인스턴스입니다.`,
            errorResponse
          );
          alert(
            "쿠키를 포함하지 않는 요청을 보내는 인스턴스 입니다. 로직을 확인하세요."
          );
          window.location.replace("/");
          break;
        case 403:
          console.error(
            `${errorStatus} Forbidden: 권한 오류 - 인증 과정이 필요한 API를 사용하면 안되는 인스턴스입니다.`,
            errorResponse
          );
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

authApi.interceptors.response.use(
  (response) => response,
  //에러 핸들링 콜백 함수임임
  async (error) => {
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

      // 401 응답 처리 (access token 만료 - 재발급 요청)
      const originalRequest = error.config;
      if (
        errorStatus === 401 &&
        !originalRequest._retry &&
        !originalRequest.url.includes("/api/v1/auth/refresh")
      ) {
        originalRequest._retry = true;
        console.error(
          `${errorStatus} Unauthorized: 인증 오류 - access token 만료 - 재발급 요청`,
          errorResponse
        );

        try {
          await authApi.post("/api/v1/auth/refresh");
          return authApi(originalRequest);
        } catch (refreshError) {
          console.error("토큰 재발급 실패, 로그인 상태 변화");

          // 로그인 페이지 리다이렉트 대신 알림 표시
          invalidRefreshToken();

          return Promise.reject(refreshError);
        }
      }

      switch (errorStatus) {
        case 400:
          console.error(`${errorStatus} 오류 입니다`, errorResponse);
          break;
        case 401:
          console.error(
            `${errorStatus} Unauthorized: 인증 오류`,
            errorResponse
          );

          // 로그인 페이지 리다이렉트 대신 알림 표시
          invalidRefreshToken();
          break;
        case 403:
          console.error(
            `${errorStatus} Forbidden: 권한 오류. 인증 되었지만 권한 없음.`,
            errorResponse
          );
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
