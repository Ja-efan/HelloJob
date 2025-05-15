package com.ssafy.hellojob.domain.coverlettercontent.repository;

import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ContentQuestionStatusDto;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import org.springframework.data.domain.Pageable;

import java.util.List;

public interface CoverLetterContentRepository extends JpaRepository<CoverLetterContent, Integer> {

    @Query("SELECT new com.ssafy.hellojob.domain.coverlettercontent.dto.response.ContentQuestionStatusDto(" +
            "clc.contentId, clc.contentNumber, clc.contentStatus) FROM CoverLetterContent clc " +
            "WHERE clc.coverLetter.coverLetterId = :coverLetterId " +
            "ORDER BY clc.contentNumber")
    List<ContentQuestionStatusDto> getCoverLetterContentStatuses(@Param("coverLetterId") Integer coverLetterId);

    @Query("""
            SELECT clc.contentId
            FROM CoverLetterContent clc
            WHERE clc.coverLetter.coverLetterId = :coverLetterId
            ORDER BY clc.contentNumber
            """)
    List<Integer> findFirstContentIdByCoverLetterId(@Param("coverLetterId") Integer coverLetterId, Pageable pageable);

    @Query("""
            SELECT clc.contentId
            FROM CoverLetterContent clc
            WHERE clc.coverLetter.coverLetterId = :coverLetterId
            ORDER BY clc.contentNumber
            """)
    List<Integer> findContentIdByCoverLetterId(@Param("coverLetterId") Integer coverLetterId);

    List<CoverLetterContent> findByCoverLetter(CoverLetter coverLetter);
}
