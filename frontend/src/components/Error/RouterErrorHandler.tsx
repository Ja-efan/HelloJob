import { useRouteError, isRouteErrorResponse, useNavigate } from "react-router";
import ErrorPage from "./ErrorPage"; // ErrorPage 공통 컴포넌트 임포트

function RouterErrorHandler() {
  const error = useRouteError();
  const navigate = useNavigate();
  const isRouteError = isRouteErrorResponse(error);

  // 기본 에러 상태 및 메시지 설정
  let status: number | string = "";
  let title: string = "오류가 발생했습니다";
  let message: string = "예상치 못한 오류가 발생했습니다.";
  let actionButton = undefined;

  if (isRouteError) {
    status = error.status;
    if (error.status === 404) {
      title = "페이지를 찾을 수 없습니다";
      message = "요청하신 URL에 해당하는 페이지가 존재하지 않습니다.";
    } else if (error.status === 401) {
      title = "접근 권한이 없습니다";
      message = "이 페이지에 접근하려면 로그인이 필요합니다.";
      actionButton = {
        label: "로그인 하기",
        onClick: () => navigate("/login"),
      };
    } else if (error.status === 403) {
      title = "접근 권한이 없습니다";
      message = "이 페이지에 접근할 권한이 없습니다.";
    } else if (error.status === 500) {
      title = "서버 오류";
      message = "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.";
    }
  }

  return (
    <ErrorPage
      status={status}
      title={title}
      message={message}
      actionButton={actionButton}
      showHomeButton={true}
      showBackButton={isRouteError ? error.status !== 401 : true}
    />
  );
}

export default RouterErrorHandler;
