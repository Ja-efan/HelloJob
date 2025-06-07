package com.ssafy.hellojob.domain.interview.repository;

import com.ssafy.hellojob.domain.interview.entity.InterviewAnswer;
import com.ssafy.hellojob.domain.interview.entity.InterviewVideo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;

@Repository
public interface InterviewAnswerRepository extends JpaRepository<InterviewAnswer, Integer> {

    List<InterviewAnswer> findInterviewAnswerByInterviewVideo(InterviewVideo interviewVideo);

    List<InterviewAnswer> findByInterviewVideoOrderByCreatedAtAsc(InterviewVideo interviewVideo);

    @Query("SELECT new map(ia.interviewVideo.interviewVideoId as videoId, ia.interviewQuestion as firstQuestion) " +
            "FROM InterviewAnswer ia " +
            "WHERE ia.interviewVideo.interviewVideoId IN :videoIds " +
            "AND ia.interviewAnswerId = (SELECT MIN(ia2.interviewAnswerId) FROM InterviewAnswer ia2 " +
            "                            WHERE ia2.interviewVideo = ia.interviewVideo)")
    List<Map<String, Object>> findFirstQuestionsByVideoIds(@Param("videoIds") List<Integer> videoIds);

    List<InterviewAnswer> findAllByInterviewVideo(InterviewVideo interviewVideo);

    @Query("SELECT COUNT(v) FROM InterviewAnswer v WHERE v.interviewVideo.interviewVideoId = :interviewVideoId")
    Long countByInterviewVideoId(@Param("interviewVideoId") Integer interviewVideoId);

    @Query("SELECT COUNT(a) FROM InterviewAnswer a WHERE a.interviewVideo.interviewVideoId = :interviewVideoId AND a.interviewAnswer IS NOT NULL")
    Long countCompletedAnswersByInterviewVideoId(@Param("interviewVideoId") Integer interviewVideoId);

    @Query("SELECT COUNT(a) FROM InterviewAnswer a WHERE a.interviewVideo.interviewVideoId = :interviewVideoId AND a.interviewVideoUrl IS NOT NULL")
    Long countSavedVideoByInterviewVideoId(@Param("interviewVideoId") Integer interviewVideoId);

    @Modifying
    @Query("""
            UPDATE InterviewAnswer ia SET ia.interviewAnswer = :answer WHERE ia.interviewAnswerId = :interviewAnswerId
            """)
    void saveInterviewAnswer(@Param("interviewAnswerId") Integer interviewAnswerId, @Param("answer") String answer);

    @Modifying
    @Query("""
            UPDATE InterviewAnswer ia SET ia.interviewAnswerFeedback = :feedback, ia.interviewFollowUpQuestion = :followUpQuestion WHERE ia.interviewAnswerId = :interviewAnswerId
            """)
    void saveInterviewFeedback(@Param("interviewAnswerId") Integer interviewAnswerId, @Param("feedback") String feedback, @Param("followUpQuestion") String followUpQuestion);

    @Modifying
    @Query("""
            UPDATE InterviewAnswer ia SET ia.interviewVideoUrl = :url, ia.videoLength = :length WHERE ia.interviewAnswerId = :id
            """)
    void saveVideoUrl(@Param("id") Integer Id, @Param("url") String url, @Param("length") String length);
}
