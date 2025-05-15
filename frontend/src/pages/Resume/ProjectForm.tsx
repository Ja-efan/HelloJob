import { PostProjectRequest } from "@/api/experienceApi";
import { Button } from "@/components/Button";
import FormInput from "@/components/Common/FormInput";
import { usePostProject } from "@/hooks/projectHooks";
import { FormEvent, useState } from "react";

interface ProjectFormProps {
  onClose: () => void;
}

function ProjectForm({ onClose }: ProjectFormProps) {
  const [formData, setFormData] = useState<PostProjectRequest>({
    projectName: "",
    projectIntro: "",
    projectRole: "",
    projectSkills: "",
    projectStartDate: "",
    projectEndDate: "",
    projectDetail: "",
    projectClient: "",
  });

  const mutation = usePostProject();

  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    mutation.mutate(formData, {
      onSuccess: () => {
        onClose();
      },
      onError: (error) => {
        console.log(error);
      },
    });
  };

  return (
    <>
      <div className="modal-overlay" onClick={handleOverlayClick}>
        <div className="modal-container h-200">
          <div className="border-b pb-3 mb-5">
            <div className="text-2xl font-bold pb-1">프로젝트 추가</div>
            <div className="text-muted-foreground text-sm">
              좀더 적합한 자소서 초안 작성을 위해 프로젝트를 추가해 주세요!
            </div>
          </div>
          <form onSubmit={handleSubmit}>
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectName"
              label="프로젝트명"
              placeholder="프로젝트명 입력"
              require
              value={formData.projectName}
              onChange={handleChange}
            />
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectIntro"
              label="개요"
              placeholder="개요 입력"
              require
              value={formData.projectIntro}
              onChange={handleChange}
            />
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectRole"
              label="역할"
              placeholder="예: 프론트엔드, 백엔드, 인프라, AI ..."
              value={formData.projectRole}
              onChange={handleChange}
            />
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectSkills"
              label="기술"
              placeholder="예: Spring boot, React, TypeScript ..."
              value={formData.projectSkills}
              onChange={handleChange}
            />
            <div className="flex gap-8">
              <FormInput
                type="date"
                width="19.5rem"
                height="2.8rem"
                name="projectStartDate"
                label="시작일"
                require
                value={formData.projectStartDate}
                onChange={handleChange}
              />
              <FormInput
                type="date"
                width="19.5rem"
                height="2.8rem"
                name="projectEndDate"
                label="종료일"
                require
                value={formData.projectEndDate}
                onChange={handleChange}
              />
            </div>
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectClient"
              label="기관"
              placeholder="예: 삼성 전자, SSAFY ..."
              value={formData.projectClient}
              onChange={handleChange}
            />
            <FormInput
              type="text"
              width="41rem"
              height="2.8rem"
              name="projectDetail"
              label="프로젝트 상세내용"
              placeholder="프로젝트 상세 내용을 입력해 주세요"
              value={formData.projectDetail}
              onChange={handleChange}
            />
            <div className="flex justify-end gap-3 mt-4">
              <Button onClick={onClose} variant={"white"} type="button">
                취소
              </Button>
              <Button type="submit">완료</Button>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default ProjectForm;
