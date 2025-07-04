package com.ssafy.hellojob.domain.interview.repository;

import com.ssafy.hellojob.domain.interview.entity.Interview;
import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface InterviewRepository extends JpaRepository<Interview, Integer> {

    Optional<Interview> findTopByUserAndCsOrderByInterviewId(User user, boolean cs);

    Optional<Interview> findByUserAndInterviewId(User user, Integer interviewId);

}
