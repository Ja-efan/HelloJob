package com.ssafy.hellojob.domain.jobrolesnapshot.repository;

import com.ssafy.hellojob.domain.jobrolesnapshot.dto.response.JobRoleSnapshotResponseDto;
import com.ssafy.hellojob.domain.jobrolesnapshot.entity.JobRoleSnapshot;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.Optional;

public interface JobRoleSnapshotRepository extends JpaRepository<JobRoleSnapshot, Integer> {

    @Query("""
            SELECT new com.ssafy.hellojob.domain.jobrolesnapshot.dto.response.JobRoleSnapshotResponseDto(
            j.jobRoleSnapshotId, 
            j.companyName, 
            j.jobRoleSnapshotName, 
            j.jobRoleSnapshotTitle, 
            j.jobRoleSnapshotSkills, 
            j.jobRoleSnapshotWork, 
            j.jobRoleSnapshotRequirements, 
            j.jobRoleSnapshotPreferences, 
            j.jobRoleSnapshotEtc, 
            j.jobRoleSnapshotCategory) 
            FROM JobRoleSnapshot j 
            WHERE j.jobRoleSnapshotId = :jobRoleSnapshotId
            """)
    Optional<JobRoleSnapshotResponseDto> findByJobRoleSnapshotId(Integer jobRoleSnapshotId);
}
