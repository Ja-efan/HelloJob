import { Button } from "@/components/Button";
import FormInput from "@/components/Common/FormInput";
import { useState } from "react";
import { toast } from "sonner";

function JobSearch() {
  const jobs = [
    [
      "서버 백엔드 개발자",
      "프론트엔드 개발자",
      "안드로이드 개발자",
      "iOS 개발자",
      "크로스 플랫폼 앱 개발자",
    ],
    [
      "게임 클라이언트 개발자",
      "게임 서버 개발자",
      "SW 솔루션",
      "VR AR 3D",
      "블록체인",
    ],
    ["DBA", "빅데이터 엔지니어", "인공지능 머신러닝", "헬스테크"],
    ["devops 시스템 엔지니어", "정보보안 침해대응", "HW 펌웨어 개발"],
    ["QA 엔지니어", "개발 PM", "기술지원", "기타"],
  ];
  const topCategory = [
    "웹/앱",
    "게임",
    "데이터&AI",
    "인프라/보안",
    "관리/지원",
  ];

  const [selectTopCategory, setSelectTopCategory] = useState(0);
  const [selectJob, setSelectJob] = useState("");
  const [isOpenInput, setIsOpenInput] = useState(false);
  const [inputJob, setInputJob] = useState("");

  const handleSelectTopCategory = (id: number) => {
    setSelectTopCategory(id);
  };

  const handleSelectJob = (jobName: string) => {
    setSelectJob(jobName.trim());
    console.log(selectJob);
  };

  const handleInputArea = () => {
    isOpenInput ? setIsOpenInput(false) : setIsOpenInput(true);
  };

  const handleInputJob = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputJob(e.target.value);
  };

  return (
    <div className=" border rounded-xl w-full p-5">
      <div className="font-semibold text-xl ">직무 선택</div>
      <div className="border flex rounded-xl my-3 ">
        {topCategory.map((item, index) => (
          <span
            onClick={() => handleSelectTopCategory(index)}
            className={`border border-transparent rounded-xl text-sm w-full h-10 flex items-center justify-center duration-100 transition-all ${
              selectTopCategory === index && "bg-primary text-white"
            }`}
          >
            {item}
          </span>
        ))}
      </div>

      <div className=" my-1">
        <p className="font-semibold mb-2"> 세부 직무</p>

        {jobs.map(
          (jobs, categoryIndex) =>
            categoryIndex === selectTopCategory && (
              <div key={categoryIndex} className="grid grid-cols-2 gap-2 ">
                {jobs.map((job, jobIndex) => (
                  <div
                    onClick={() => handleSelectJob(job)}
                    key={jobIndex}
                    className={`border flex justify-center items-center h-10 rounded-lg hover:border-primary hover:border-[1.5px] hover-block
                    ${
                      selectJob === job &&
                      "bg-hover-block border-primary border-[1.5px]"
                    }
                    `}
                  >
                    {job}
                  </div>
                ))}
              </div>
            )
        )}
      </div>
      <div
        onClick={handleInputArea}
        className="cursor-pointer mt-20 w-35 font-semibold text-text-muted-foreground"
      >
        찾는 직무가 없어요..
      </div>
      {isOpenInput && (
        <div className="flex  items-center">
          <FormInput
            placeholder="원하시는 직무를 직접 입력해 주세요"
            width="20rem"
            type="text"
            height="2.5rem"
            name={"category"}
            onChange={handleInputJob}
            className="w-full"
          />
          <Button
            className="ml-1 h-10 w-14"
            onClick={() => {
              handleSelectJob(inputJob);
              toast.info("저장되었습니다.");
            }}
          >
            저장
          </Button>
        </div>
      )}
    </div>
  );
}

export default JobSearch;
