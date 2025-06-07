package com.ssafy.hellojob.domain.interview.repository;

import com.ssafy.hellojob.domain.interview.entity.InterviewVideo;
import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface InterviewVideoRepository extends JpaRepository<InterviewVideo, Integer> {

    @Query("SELECT iv FROM InterviewVideo iv " +
            "LEFT JOIN iv.interview i " +
            "LEFT JOIN iv.coverLetterInterview cli " +
            "WHERE i.user = :user OR cli.user = :user " +
            "ORDER BY iv.start DESC NULLS LAST")
    List<InterviewVideo> findAllByUser(@Param("user") User user);


    @Query("SELECT iv FROM InterviewVideo iv " +
            "LEFT JOIN FETCH iv.coverLetterInterview " +
            "LEFT JOIN FETCH iv.interview " +
            "WHERE iv.interviewVideoId = :id")
    Optional<InterviewVideo> findByIdWithInterviewAndCoverLetterInterview(@Param("id") Integer interviewVideoId);


    List<InterviewVideo> findAllByStartBeforeAndInterviewTitleIsNull(LocalDateTime cutoff);

    @Query("SELECT COUNT(ia) FROM InterviewAnswer ia WHERE ia.interviewVideo.interviewVideoId = :videoId")
    Integer countTotalAnswer(@Param("videoId") Integer videoId);

    @Query("SELECT COUNT(ia) FROM InterviewAnswer ia WHERE ia.interviewVideo.interviewVideoId = :videoId AND ia.interviewAnswer IS NOT NULL")
    Integer countSavedAnswer(@Param("videoId") Integer videoId);

    @Modifying
    @Query("""
            UPDATE InterviewVideo iv SET iv.interviewFeedback = :feedback, iv.feedback = true WHERE iv.interviewVideoId = :videoId
            """)
    void saveFeedback(@Param("videoId") Integer videoId, @Param("feedback") String feedback);

    @Modifying
    @Query("""
            UPDATE InterviewVideo iv SET iv.interviewTitle = :title, iv.end = now() WHERE iv.interviewVideoId = :id
            """)
    void saveTitle(@Param("id") Integer id, @Param("title") String title);
}
