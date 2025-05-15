import { authApi } from "./instance";

export interface PostProjectRequest {
  projectName: string;
  projectIntro: string;
  projectRole: string;
  projectSkills: string;
  projectStartDate: string;
  projectEndDate: string;
  projectDetail: string;
  projectClient: string;
}

export interface PostProjectResponse {
  projectId: number;
  message: string;
}

export interface GetProjectsResponse {
  projectId: number;
  projectName: string;
  projectIntro: string;
  projectSkills: string;
  updatedAt: string;
}

export interface GetProjectResponse {
  projectId: number;
  projectName: string;
  projectIntro: string;
  projectRole: string;
  projectSkills: string;
  projectStartDate: string;
  projectEndDate: string;
  projectDetail: string;
  projectClient: string;
  updatedAt: string;
}
export const experienceApi = {
  postProject: (projectFormData: PostProjectRequest) => {
    return authApi.post("/api/v1/project", projectFormData);
  },
  getProjects: () => {
    return authApi.get("/api/v1/project");
  },
  getProject: (projectId: number) => {
    return authApi.get(`/api/v1/project/${projectId}`);
  },
  putProject: (projectFormData: PostProjectRequest, projectId: number) => {
    return authApi.put(`/api/v1/project/${projectId}`, projectFormData);
  },
  deleteProject: (projectId: number) => {
    return authApi.delete(`/api/v1/project/${projectId}`);
  },
};
