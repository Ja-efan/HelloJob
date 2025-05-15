package com.ssafy.hellojob.domain.schedule.repository;

import com.ssafy.hellojob.domain.schedule.entity.ScheduleStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ScheduleStatusRepository extends JpaRepository<ScheduleStatus, Integer> {

    Optional<ScheduleStatus> findByScheduleStatusName(String scheduleStatusName);

}
