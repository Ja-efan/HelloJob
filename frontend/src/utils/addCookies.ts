import { useEffect } from "react";

export const addCookies = () => {
  useEffect(() => {
    const accessToken = import.meta.env.VITE_DEV_ACCESS_TOKEN as string;
    const refreshToken = import.meta.env.VITE_DEV_REFRESH_TOKEN as string;

    setDevAuthCookie(accessToken, refreshToken);
  }, []);

  const setDevAuthCookie = (accessToken: string, refreshToken: string) => {
    if (process.env.NODE_ENV !== "development") return;
    document.cookie = `refresh_token=${refreshToken}; path=/; max-age=3600`;
    document.cookie = `access_token=${accessToken}; path=/; max-age=3600`;
    console.log("개발용 인증 쿠키가 설정되었습니다.");
  };
};
