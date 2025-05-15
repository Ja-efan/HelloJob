package com.ssafy.hellojob.domain.schedule.repository;

import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.schedule.dto.response.ScheduleListResponseDto;
import com.ssafy.hellojob.domain.schedule.entity.Schedule;
import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ScheduleRepository extends JpaRepository<Schedule, Integer> {

    Optional<Schedule> findByCoverLetter(CoverLetter coverLetter);

    List<Schedule> findByUser(User user);


}
