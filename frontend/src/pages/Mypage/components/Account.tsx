import MypageHeader from "./MypageHeader";

function Account() {
  return (
    <div className="flex-1 p-4 md:p-6 md:ml-56 transition-all duration-300">
      <MypageHeader title="계정 설정" />
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">계정 설정</h2>
        <div className="bg-gray-50 p-4 rounded-md text-center">
          <p>계정 정보를 관리할 수 있는 페이지입니다.</p>
          <p>3차 배포에서 토큰 관리 기능을 추가할 예정입니다.</p>
        </div>
      </div>
    </div>
  );
}

export default Account;
