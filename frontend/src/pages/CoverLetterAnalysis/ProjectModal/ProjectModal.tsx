import { Button } from "@/components/Button";

// {
//     "projectId": 8,
//     "projectName": "걍 입력해봄",
//     "projectIntro": "걍 입력해봤습니다",
//     "projectSkills": "SpringBoot, JPA, mySQL",
//     "updatedAt": "2025-04-28T15:35:43.887088"
// },

function ProjectModal() {
  //   const { data } = useGetProjects();
  //   console.log(data);
  const data = [
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
    {
      projectId: 8,
      projectName: "걍 입력해봄",
      projectIntro: "걍 입력해봤습니다",
      projectSkills: "SpringBoot, JPA, mySQL",
      updatedAt: "2025-04-28T15:35:43.887088",
    },
  ];

  const calculateDaysAgo = (dateString: string) => {
    const updatedDate = new Date(dateString);
    const currentDate = new Date();
    const diffTime = currentDate.getTime() - updatedDate.getTime();
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    return diffDays === 0 ? "오늘" : `${diffDays}일 전`;
  };
  return (
    <div className="modal-overlay">
      <div className="modal-container h-150 w-150">
        <div className="text-2xl font-bold mb-2">내 프로젝트</div>
        <div className="">자소서 생성에 활용할 프로젝트를 선택해주세요</div>
        <hr className="mb-3" />
        <div className="overflow-auto h-100 border-y">
          {data.map((el) => (
            <div
              key={el.projectId}
              className="  cursor-pointer border border-l-4 border-l-primary h-15 my-2 rounded-2xl flex items-center justify-between px-5 hover-block"
            >
              <span className="font-semibold">{el.projectName}</span>
              <span>{el.projectIntro}</span>
              <span>{calculateDaysAgo(el.updatedAt)}</span>
            </div>
          ))}
        </div>
        <div className="mt-5 flex justify-end gap-3">
          <Button variant={"white"} className="w-15">
            취소
          </Button>
          <Button className="w-15">완료</Button>
        </div>
      </div>
    </div>
  );
}

export default ProjectModal;
