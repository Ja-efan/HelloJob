import MypageHeader from "./MypageHeader";

function InterviewVideos() {
  return (
    <div className="flex-1 p-4 md:p-6 md:ml-56 transition-all duration-300">
      <MypageHeader title="모의 면접 영상" />
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold mb-4">모의 면접 영상</h2>
        <div className="bg-gray-50 p-4 rounded-md text-center">
          <p>저장된 모의 면접 영상을 확인할 수 있습니다.</p>
          <p>2차 배포에서 추후 추가 될 기능입니다.</p>
        </div>
      </div>
    </div>
  );
}

export default InterviewVideos;
