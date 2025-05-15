import MypageHeader from "./MypageHeader";

function BookmarkedJobs() {
  return (
    <div className="flex-1 p-4 md:p-6 md:ml-56 transition-all duration-300">
      <MypageHeader title="직무 분석 즐겨찾기" />
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">직무 분석 목록</h2>
        <div className="bg-gray-50 p-4 rounded-md text-center">
          <p>북마크한 직무 목록을 확인할 수 있습니다.</p>
          <p>관심 있는 직무를 한눈에 관리해 보세요.</p>
        </div>
      </div>
    </div>
  );
}

export default BookmarkedJobs;
