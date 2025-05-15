package com.ssafy.hellojob.domain.coverletter.repository;

import com.ssafy.hellojob.domain.coverletter.dto.response.MyPageCoverLetterDto;
import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;


public interface CoverLetterRepository extends JpaRepository<CoverLetter, Integer> {
    @Modifying
    @Transactional
    @Query("UPDATE CoverLetter c SET c.updatedAt = CURRENT_TIMESTAMP WHERE c.coverLetterId = :coverLetterId")
    void touch(@Param("coverLetterId") Integer coverLetterId);

    @Query("""
            SELECT new com.ssafy.hellojob.domain.coverletter.dto.response.MyPageCoverLetterDto(
            cl.coverLetterId, cl.coverLetterTitle, c.companyName, jr.jobRoleSnapshotName, jr.jobRoleSnapshotCategory, cl.finish, cl.updatedAt)
            FROM CoverLetter cl
            JOIN cl.jobRoleSnapshot jr
            JOIN cl.companyAnalysis ca
            JOIN ca.company c
            WHERE cl.user.userId = :userId
            """)
    Page<MyPageCoverLetterDto> getCoverLettersByUser(@Param("userId") Integer userId, Pageable pageable);
}
