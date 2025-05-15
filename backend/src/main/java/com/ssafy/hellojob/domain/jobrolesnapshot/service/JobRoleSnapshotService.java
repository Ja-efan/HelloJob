package com.ssafy.hellojob.domain.jobrolesnapshot.service;

import com.ssafy.hellojob.domain.jobrolesnapshot.dto.response.JobRoleSnapshotResponseDto;
import com.ssafy.hellojob.domain.jobrolesnapshot.entity.JobRoleSnapshot;
import com.ssafy.hellojob.domain.jobrolesnapshot.repository.JobRoleSnapshotRepository;
import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysis;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
public class JobRoleSnapshotService {

    private final JobRoleSnapshotRepository jobRoleSnapshotRepository;

    public JobRoleSnapshot copyJobRoleAnalysis(String companyName, JobRoleAnalysis jobRoleAnalysis) {

        JobRoleSnapshot jobRoleSnapshot = JobRoleSnapshot.builder()
                .companyName(companyName)
                .jobRoleSnapshotName(jobRoleAnalysis.getJobRoleName())
                .jobRoleSnapshotTitle(jobRoleAnalysis.getJobRoleTitle())
                .jobRoleSnapshotSkills(jobRoleAnalysis.getJobRoleSkills())
                .jobRoleSnapshotCategory(jobRoleAnalysis.getJobRoleCategory().toString())
                .jobRoleSnapshotPreferences(jobRoleAnalysis.getJobRolePreferences())
                .jobRoleSnapshotRequirements(jobRoleAnalysis.getJobRoleRequirements())
                .jobRoleSnapshotEtc(jobRoleAnalysis.getJobRoleEtc())
                .jobRoleSnapshotWork(jobRoleAnalysis.getJobRoleWork())
                .build();

        jobRoleSnapshotRepository.save(jobRoleSnapshot);

        return jobRoleSnapshot;
    }

    @Transactional
    public JobRoleSnapshotResponseDto getJobRoleSnapshot(Integer jobRoleSnapshotId) {
        return jobRoleSnapshotRepository.findByJobRoleSnapshotId(jobRoleSnapshotId)
                .orElseThrow(() -> new BaseException(ErrorCode.JOB_ROLE_SNAPSHOT_NOT_FOUND));
    }
}
