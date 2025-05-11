import MypageHeader from "./MypageHeader";

function Schedule() {
  return (
    <div className="flex-1 p-4 md:p-6 md:ml-56 transition-all duration-300">
      <MypageHeader title="일정 관리" />
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">일정 관리</h2>
        <div className="bg-gray-50 p-4 rounded-md text-center">
          <p>일정 관리 페이지입니다.</p>
          <p>2차 배포에서 추후 추가 될 기능입니다.</p>
        </div>
      </div>
    </div>
  );
}

export default Schedule;
