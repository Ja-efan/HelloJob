import { useEffect } from "react";
import { Outlet, useNavigate } from "react-router";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { addCookies } from "./utils/addCookies";
import { useAuthStore } from "./store/userStore";
import { userAuthApi } from "./api/userAuth";

function App() {
  addCookies();
  const navigate = useNavigate();
  //isLoginAlert는 로그인 페이지로 강제 이동하는 기능을 대체하는 역할이다.
  const { isLoginAlert, setIsLoggedIn, setUserName, setIsLoginAlert } =
    useAuthStore();

  useEffect(() => {
    const checkLogin = async () => {
      try {
        //error가 발생하지 않고 정보가 불러와지면 로그인 한 것.
        const res = await userAuthApi.getLoginUserInfo();
        setIsLoggedIn(true);
        setUserName(res.data.nickname);
        console.log("로그인 성공");
      } catch (error) {
        setIsLoggedIn(false);
      }
    };

    checkLogin();
  }, [setIsLoggedIn, setUserName]);

  useEffect(() => {
    if (isLoginAlert) {
      toast("현재 로그아웃 상태입니다", {
        id: "login-alert",
        description: "다양한 기능 사용을 위해 로그인 해주세요.",
        action: {
          label: "로그인",
          onClick: () => navigate("/login"),
        },
      });
      setIsLoginAlert(false);
    }
  }, [isLoginAlert, setIsLoginAlert, navigate]);

  return (
    <div id="wrap" className="bg-background min-h-screen flex flex-col">
      <Outlet />
      <Toaster />
    </div>
  );
}

export default App;
