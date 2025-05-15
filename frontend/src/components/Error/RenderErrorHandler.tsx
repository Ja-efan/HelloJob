import { RenderErrorHandlerProps } from "@/types/errorTypes";
import ErrorPage from "./ErrorPage";

function RenderErrorFallback({
  error,
  resetErrorBoundary,
}: RenderErrorHandlerProps) {
  return (
    <ErrorPage
      title="컴포넌트 렌더링 오류"
      message={error.message || "컴포넌트를 표시하는 중 문제가 발생했습니다."}
      actionButton={{
        label: "다시 시도",
        onClick: resetErrorBoundary,
      }}
    ></ErrorPage>
  );
}

export default RenderErrorFallback;
