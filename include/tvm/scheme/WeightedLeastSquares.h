/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/meta.h>
#include <tvm/scheme/abstract/ResolutionScheme.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/LinearizedProblemComputationData.h>
#include <tvm/solver/abstract/LeastSquareSolver.h>

// Creating a class tvm::internal::has_member_type_Factory<T>
TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(Factory)

namespace tvm
{

namespace scheme
{
/** A set of options for WeightedLeastSquares. */
class TVM_DLLAPI WeightedLeastSquaresOptions
{
  /** If \a true, a damping task is added when no constraint with level >=1 has been
   * given.
   */
  TVM_ADD_NON_DEFAULT_OPTION(autoDamping, false)
  /**  The factor to emulate priority for priority levels >= 1.
   * E.g. if a task T1 has a weight w1 and priority 1, and a task T2 has a weight w2 and
   * priority 2, the weighted least-squares problem will be assembled with weights
   * \p scalarizationWeight * w1 and w2 for T1 and T2 respectively. */
  TVM_ADD_NON_DEFAULT_OPTION(scalarizationWeight, 1000.)
};

/** This class implements the classic weighted least square scheme. */
class TVM_DLLAPI WeightedLeastSquares : public abstract::LinearResolutionScheme<WeightedLeastSquares>
{
private:
  struct Memory : public internal::LinearizedProblemComputationData
  {
    Memory(int solverId, std::unique_ptr<solver::abstract::LeastSquareSolver> solver);

    std::unique_ptr<solver::abstract::LeastSquareSolver> solver;

    int maxp;

  protected:
    void setVariablesToSolution_(tvm::internal::VariableCountingVector & x) override;
  };

  const static internal::SchemeAbilities abilities_;

  /** Check if T derives from LSSolverFactory. */
  template<typename T>
  using isFactory = std::is_base_of<solver::abstract::LSSolverFactory, T>;
  /** Helper struct for isOption .*/
  template<typename T, bool>
  struct isOption_ : std::false_type
  {};
  /** Helper struct specialization for isOption .*/
  template<typename T>
  struct isOption_<T, true>
  {
    static const bool value = isFactory<typename T::Factory>::value;
  };
  /** Check if T has a member T::Factory and if so if T::Factory derives from LSSolverFactory.*/
  template<typename T>
  using isOption = isOption_<T, tvm::internal::has_member_type_Factory<T>::value>;

public:
  using ComputationDataType = Memory;

  /** Constructor from a LSSolverFactory
   * \tparam SolverFactory Any class deriving from LSSolverFactory.
   * \param solverFactory A configuration for the solver to be used by the resolution scheme.
   * \param schemeOptions Options for the schemes. See tvm::Scheme::WeightedLeastSquaresOptions.
   */
  template<class SolverFactory, typename std::enable_if<isFactory<SolverFactory>::value, int>::type = 0>
  WeightedLeastSquares(const SolverFactory & solverFactory, WeightedLeastSquaresOptions schemeOptions = {})
  : LinearResolutionScheme<WeightedLeastSquares>(abilities_), options_(schemeOptions),
    solverFactory_(solverFactory.clone())
  {}

  /** Constructor from a configuration class
   * \tparam SolverOptions Any class representing solver options. The class must have a
   *    member type \a Factory referring to a class C deriving from LSSolverFactory
   *    and such that C can be constructed from SolverOptions.
   * \param solverOptions A set of options for the solver to be used by the resolution scheme.
   * \\param schemeOptions Options for the scheme. See tvm::Scheme::WeightedLeastSquaresOptions.
   */
  template<class SolverOptions, typename std::enable_if<isOption<SolverOptions>::value, int>::type = 0>
  WeightedLeastSquares(const SolverOptions & solverOptions, WeightedLeastSquaresOptions schemeOptions = {})
  : WeightedLeastSquares(typename SolverOptions::Factory(solverOptions), schemeOptions)
  {}

  /** A fallback constructor that is enabled when none of the others are.
   * It always fails at compilation time to provide a nice error message.
   */
  template<typename T, typename std::enable_if<!isFactory<T>::value && !isOption<T>::value, int>::type = 0>
  WeightedLeastSquares(const T &, WeightedLeastSquaresOptions = {})
  : LinearResolutionScheme<WeightedLeastSquares>(abilities_)
  {
    static_assert(tvm::internal::always_false<T>::value,
                  "First argument can only be a LSSolverFactory or a solver configuration. "
                  "A configuration needs to have a Factory member type that is itself deriving from LSSolverFactory. "
                  "See LSSOLLSSolverOptions for an example.");
  }

  /** \internal Copy and move are deleted because of the unique_ptr members of
   * the class on polymorphic types. If these semantics are needed, it should
   * be possible to implement them with good care.
   */
  WeightedLeastSquares(const WeightedLeastSquares &) = delete;
  WeightedLeastSquares(WeightedLeastSquares &&) = delete;
  WeightedLeastSquares & operator=(const WeightedLeastSquares &) = delete;
  WeightedLeastSquares & operator=(WeightedLeastSquares &&) = delete;

  /** Private interface for CRTP*/
  bool solve_(const LinearizedControlProblem & problem, internal::ProblemComputationData * data) const;
  void updateComputationData_(const LinearizedControlProblem & problem, internal::ProblemComputationData * data) const;
  std::unique_ptr<Memory> createComputationData_(const LinearizedControlProblem & problem) const;

protected:
  void addTask(const LinearizedControlProblem & problem,
               Memory * memory,
               TaskWithRequirements * task,
               solver::internal::SolverEvents & se) const;
  void removeTask(const LinearizedControlProblem & problem,
                  Memory * memory,
                  TaskWithRequirements * task,
                  solver::internal::SolverEvents & se) const;

  WeightedLeastSquaresOptions options_;
  /** The factory to create solvers attached to each problem. */
  std::unique_ptr<solver::abstract::LSSolverFactory> solverFactory_;
};

} // namespace scheme

} // namespace tvm
