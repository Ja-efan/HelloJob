import { PostProjectRequest } from "./experienceApi";
import { authApi } from "./instance";

export const projectApi = {
  getProjects: () => {
    return authApi.get("/api/v1/project");
  },
  getProject: (projectId: number) => {
    return authApi.get(`/api/v1/project/${projectId}`);
  },
  postProject: (projectFormData: PostProjectRequest) => {
    return authApi.post("/api/v1/project", projectFormData);
  },
  putProject: (projectFormData: PostProjectRequest, projectId: number) => {
    return authApi.put(`/api/v1/project/${projectId}`, projectFormData);
  },
  deleteProject: (projectId: number) => {
    return authApi.delete(`/api/v1/project/${projectId}`);
  },
};
