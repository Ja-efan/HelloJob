import { Button } from "@/components/Button";
import { Link } from "react-router";
import googleIconUrl from "@/assets/google-icon.svg";
function Login() {
  const handleGoogleLogin = () => {
    window.location.replace(
      "https://k12b105.p.ssafy.io/oauth2/authorization/google"
    );
  };
  return (
    <>
      <div className="container relative h-screen flex-col items-center justify-center lg:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
        <div className="relative hidden h-full flex-col bg-muted p-10 text-white dark:border-r lg:flex">
          <div className="absolute inset-0 bg-zinc-900"></div>
          <div className="relative z-20 mt-auto">
            <blockquote className="space-y-2">
              <p className="typing-demo text-4xl font-gothic">
                print("<span className="text-[#AF9BFF]">Hello Job!</span>")
              </p>
              <p className="typing-demo-2 text-4xl font-gothic">
                취업 준비도 개발처럼, 체계적으로.
              </p>
            </blockquote>
          </div>
          <div className="relative z-20 mt-auto"></div>
          <div className="relative z-20 mt-auto">
            <blockquote className="space-y-2">
              <p className="text-lg whitespace-normal font-gothic">
                SSAFY 12기 B105 자율 Project
              </p>
            </blockquote>
          </div>
        </div>

        <div className="p-4 lg:p-8">
          <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
            <div className="flex flex-col space-y-2 text-center">
              <h1 className="text-2xl font-semibold tracking-tight">
                로그인하고 시작하기
              </h1>
              <p className="text-sm text-muted-foreground whitespace-normal break-keep">
                간편하게 로그인하고 기업분석, 자기소개서, 면접까지 모두
                해결하세요.
              </p>
            </div>

            <div className="grid gap-6">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t"></span>
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">
                    다음으로 계속하기
                  </span>
                </div>
              </div>
              <div className="grid gap-2">
                <Button
                  variant="white"
                  size="default"
                  type="button"
                  className="inline-flex items-center"
                  onClick={handleGoogleLogin}
                >
                  <img
                    src={googleIconUrl}
                    alt="Google 로고"
                    className="h-4 w-4"
                  />
                  Google 로그인
                </Button>
              </div>
            </div>
            <p className="px-8 text-center text-sm text-[#6E7180] whitespace-normal break-keep">
              계속하면{" "}
              <Link
                to="/"
                className="underline underline-offset-4 hover:text-[#2A2C35]"
              >
                서비스 약관
              </Link>
              과{" "}
              <Link
                to="/"
                className="underline underline-offset-4 hover:text-[#2A2C35]"
              >
                개인정보 정책
              </Link>
              에 동의하게 됩니다.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

export default Login;
